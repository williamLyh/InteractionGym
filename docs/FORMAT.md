# InteractionGym trajectory format (schema v1)

The **standard trajectory format** the env writes after an episode. The machine-checkable definition is [trajectory.schema.json](trajectory.schema.json); a complete example is [format_example.json](format_example.json).

## 0. Implementation

| Function | Location |
|---|---|
| Build an episode from an Env | `interaction_gym.traj.episode(env, episode_id, run_id=, media=, reward=, meta=)` |
| Read / write (one JSONL file per run) | `traj.save(episodes, path, append=False)` / `traj.load(path)` |
| Restore the runtime event log | `traj.to_frames(ep, media=None)` |
| Derive the agent's view | `traj.agent_view(ep, chunk_ms)` |
| Shared media store | `interaction_gym.media.MediaStore(root)`: `put` / `ref` / `load` |
| Derive `eval.duplex` | `interaction_gym.eval.duplex(turns)`; works on any loaded episode |
| Viewer | `interaction_gym.viewer.export_html(episodes, path)`, or `python -m interaction_gym.viewer runs/*.jsonl -o viewer.html` |

`kind` is written by the producer at generation time: `UserTurn(kind=...)` → `Segment.kind` → the turn's `kind`.

## 1. Design principles

- **Organized by timeline**: the body of an episode is `turns`, sorted by start time. Everything related to a turn (e.g. tool calls) is attached to that turn.
- **Facts separate from judgments**: `turns` records only what happened; `eval` is a judgment derived from `turns`. Changing the evaluation rules means recomputing `eval`; `turns` stays unchanged.
- **Annotate only what cannot be derived**: interruptions, yielding and response latency all follow from timing and are not written into turns. Only "what type of sound is this" (`kind`) must be annotated by the producer.
- **Reference large files, do not embed them**: audio and video live in a shared media store; the trajectory holds only references, including the start/end position within the file.
- **Runtime and storage are separate**: the Sim's runtime log is an append-only event stream (for causality, real-time consumption, forking and replay); it is merged into this format when saved. The two can be converted into each other (see §7).

## 2. File organization

| Level | Form |
|---|---|
| One episode | One JSON object (the structure defined in this document) |
| One run (a batch of rollouts / one evaluation) | One JSONL file, **one episode per line**, e.g. `runs/<run_id>/episodes.jsonl`; may be compressed or sharded |
| Media | A shared store directory (`meta.media_root`), content-addressed: `media/<first two hash chars>/<sha256>.<ext>`; identical content is stored once |
| Debugging a single episode | Can be exported as an indented `.json` file containing the same object |

Per-step agent observations are not saved separately: given the agent's `chunk_ms`, they can be derived deterministically from `turns` (see §7). Model-side tokens and logprobs are saved by the RL framework itself and aligned via `meta.episode_id` and the step index.

## 3. Conventions

- **Time**: all times are integers in milliseconds, with the episode start at 0. Field names ending in `_time` (`start_time`, `end_time`, `call_time`, `result_time`) are instants on the timeline.
- **Position within media**: `start_ms` / `end_ms` are positions **inside the media file**, which is distinct from instants on the timeline.
- **Role**: `role` is `user` or `agent`. Multi-party conversations may use other names, together with the `to` field.
- **Optional fields**: **omit** a field when it does not apply; do not write `null` (except where `eval` explicitly allows `null`).

## 4. Top-level structure

```json
{
  "schema": 1,
  "meta": { … },
  "background": [ … ],
  "turns": [ … ],
  "eval": { … }
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema` | integer | yes | Format version, currently `1` |
| `meta` | object | yes | Episode settings and attributes |
| `background` | array | no | Background audio tracks spanning the timeline |
| `turns` | array | yes | The timeline, sorted by `start_time` |
| `eval` | object | no | Evaluation results (omitted when not evaluated) |

### 4.1 `meta`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `episode_id` | string | yes | Unique episode identifier |
| `run_id` | string | no | The run this episode belongs to |
| `seed` | integer | no | Random seed |
| `media_root` | string | when there are media references | Root directory of the media store; every `media.uri` is relative to it |
| `task` | object | yes | The task: `id`, `scenario` (persona, instructions, first turn, etc.), `initial_state`, `criteria` |
| `env` | object | yes | Environment context: which components the environment consists of, and what the environment gives the agent (tool schemas, instructions, context); see §4.1.1 |
| `user` | object | when there is a user simulator | Details of the user simulation: persona, goal, voice, models used, barge-in and turn-taking policy; see §4.1.2 |
| `agent` | object | no (filled by runners) | The agent the episode ran: `{name, model, kind, server, ...}`. `kind`: `full-duplex` (one model that listens and speaks at once, e.g. MiniCPM-o 4.5 on vLLM-Omni) / `cascaded` (ASR → LLM → TTS) / `text` (a scripted or LLM text agent timed by a speaking rate). Agents describe themselves (`describe()`: `VllmOmniDuplexAgent` adds `clock`, `audio_out`; `CascadedAgent` its ASR / LLM / TTS) and runners pass them as `traj.episode(..., agent=agent)`; extra keys are agent-specific settings (e.g. sampling). The viewer shows it in the episode header |
| `duration_ms` | integer | yes | Total length of the timeline |
| `end_reason` | string | yes | How the episode ended: `idle` (natural end, see below) / `max_duration` (maximum duration reached; speech still in progress is cut at that instant) |

**How an episode ends**: "ending" is not simulated as an action. After its last utterance the user stops talking; all speech in progress (including the agent's reply) plays to completion, **neither side is cut off**; once a period with no new activity has passed (default 3000 ms, `Env(end_idle_ms=…)`), the episode ends naturally (`end_reason: "idle"`). Only reaching `Env(max_ms=…)` forces truncation (`end_reason: "max_duration"`).

### 4.1.1 `meta.env`: environment context

Records what the environment consists of, and **what the environment gives the agent at t=0** (later additions are recorded here too). The environment provides only instructions, tool schemas (or a directory for finding tools) and context the task explicitly allows; it **never provides world state** (e.g. database contents, which the agent must query through tools).

| Field | Type | Meaning |
|---|---|---|
| `components` | object | The components the environment consists of: `{component name: type}`, e.g. `{"user": "UserSim", "tools": "ToolWorld"}` (user simulator, tool environment; later possibly a physics simulation, a mixer, etc.) |
| `tool_schema_mode` | string | How tool schemas are provided. `all`: the schemas of all tools are given at t=0 (benchmark setting); `discover`: only a service directory and three meta-tools `list_services` / `search_tools` / `load_tools` are given, and the agent must find and load tools itself (realistic setting with many tools) |
| `instructions` | string | Instructions given to the agent at t=0 (e.g. domain rules); in `discover` mode, instructions on how to find tools; each service's own rules are provided when it is loaded |
| `tools` | array | Tools the agent can call at t=0, with **full schemas**: `{name, description, parameters, service}`; `parameters` is the JSON Schema of the arguments (as in OpenAI function calling) |
| `tool_services` | array | Service directory in `discover` mode: `{service, description, tools}` (`tools` is the number of tools in that service) |
| `agent_context` | object | Context given to the agent at t=0, from two sources: information the task allows the agent to know (`task.scenario["agent_context"]`, e.g. the caller's phone number), and **information the tool environment itself chooses to reveal** (`ToolBackend.context()`, e.g. an account summary; secret information that requires a tool call to obtain is excluded by the tool environment). Keyed by service name when several services are mounted |
| `tool_updates` | array | Tools added later: `{time, tools, instructions, context}`. `time` is when the agent receives the update, `tools` are the full schemas of the new tools, `instructions` appears only when the instructions change, `context` is what the newly loaded service reveals (in `discover` mode a service's context is revealed only when it is loaded) |

In `discover` mode, finding and loading tools are ordinary tool calls, recorded in the `tool_calls` of the corresponding turn (with the same virtual latency); the instant loading completes appears in `tool_updates`. When several services are mounted at once, tool names are written as `<service>__<tool>`.

The agent's update granularity (`chunk_ms`) is part of the agent-side run configuration and is not written into the trajectory; to derive what the agent received at each step, call `traj.agent_view(ep, chunk_ms)` with it explicitly.

### 4.1.2 `meta.user`: user simulation

Records **who the simulated user in this episode is and how it was generated**. Only the user simulation component knows this (which LLM, which TTS voice, how it decides to barge in), so the component provides it itself (`profile(task)`), following the same principle as `kind` being annotated by the producer. Persona and goal are also in `task.scenario`; this section collects the values actually in effect (e.g. `voice.speaker` is the voice specified by the task, `scenario["voice"]`, or the component's default voice when none is specified).

| Field | Type | Meaning |
|---|---|---|
| `component` | string | Name of the user simulation component (its key in `meta.env.components`) |
| `mode` | string | `offline`: replays turns with predetermined timing, regardless of what the agent does (`ReplayUser`); `semi_online`: fixed content, timing reacts to the conversation (`UserSim` + `ScriptSource`); `online`: content generated by an LLM from the persona and the conversation it hears (`UserSim` + `LLMSource`) |
| `persona` | string | User persona (`task.scenario["persona"]`); omitted when absent |
| `profile` | object | Structured user profile (`task.scenario["profile"]`): `name`, `gender` (female/male), `age` (years, or child/young/adult/senior), `language` (the language spoken in this conversation, e.g. `en`; the user LLM is told to speak only this language, and it is also the TTS `language`), `native_language` (native language; affects accent and voice selection; defaults to `language`), `accent`, `occupation`, `speaking_style`, `traits`, etc. Used to pick the voice, for the TTS style instructions and for the persona description given to the user LLM. `attentiveness`, `backchanneling`, `surroundings` and `hesitancy` determine the online user's habits (see `behaviors`) |
| `goal` | string | The user's goal (`task.scenario["instructions"]`); omitted when absent |
| `voice` | object | Voice: `{speaker, chosen_by, style, instructions, clone, leveling, seed, tts: {model, sr}}`. `chosen_by`: `task` (specified by the task, `scenario["voice"]`) / `profile` (picked from the voice catalog by profile: gender must match; same age group +3, adjacent +1; language +1; accent +1; ties broken deterministically by a hash of the name) / `default`. `style` says where the TTS `instructions` come from: `neutral` (the default since 2026-10-08: `user.NEUTRAL_STYLE`, "Natural, relaxed everyday phone-call voice at a normal pace; not acted, not dramatic." — the persona picks the voice and the words, not an acted delivery), `persona` (`profile.speaking_style` or `scenario["voice_style"]`, the earlier behaviour), or a custom text. When `clone` is present (the default whenever a TTS is used), the whole conversation uses one voice: the first utterance with content (at least 3 words) serves as the reference, and every later turn is cloned from it with `clone.tts` (which carries its voice, pace and manner over; cloned turns take no style instruction); its absence means the runner opted out (`Voice(clone=False)`) and every turn was synthesized on its own. `leveling` (default on): every synthesized turn is trimmed of leading / trailing silence (keeping `trim_margin_ms`) and normalised to an active-speech RMS of `level_dbfs` plus the persona's fixed `offset_db` (±`persona_offset_db` for a quiet / loud speaking style, else a small offset from a hash of the name), peak-limited to `peak_dbfs`. `seed`: the sampling seed sent with every TTS request of the episode (from the task id and speaker; only to TTS clients that take one). `tts: null` means no speech was synthesized, only text, with durations estimated from `words_per_sec`; `recorded_turns` is the number of turns that used recorded audio directly |
| `llm` | object | In `online` mode, the model that generates user utterances: `{model, params}` (`params` are sampling parameters, e.g. `max_tokens`) |
| `system_prompt` | string | Present only when a custom user prompt template was used (the default template is `user.USER_SYSTEM`) |
| `barge_in` | object | Who may decide that the user cuts in: `{type: "never"}` / `{type: "keyword", words}` (a rule) / `{type: "llm", llm, min_words, params}` (the listener model, below). Absent in `offline` mode |
| `listening` | object | How the user's **subjective** sounds are decided while the agent speaks (backchannels, barge-ins): `decision_points` (`"phrase boundaries"`: the end of each punctuation-delimited phrase of the agent's transcript, at the instant the user has heard it, mapped linearly in time like the heard text), `max_decision_gap_ms` (a fallback point when the agent goes this long without a boundary), `model` (the listener LLM: `{type, llm, min_words, params}` with greedy `params`, or `null`: no model, the user only listens or follows its rule) and `options` (the choices the model gets: `LISTEN`, `BACKCHANNEL` unless `backchanneling: none` (offered at phrase boundaries only, never at a fallback point), `INTERRUPT` only when the model is also `barge_in`). One call per decision point returns the choice (plus the sound); why the user cuts in is not asked here: on `INTERRUPT` the user LLM, writing the barge-in line in the same single generation call, is told it is cutting in and picks the `intent` itself (`user.BARGE_IN_NOTE`: a first line `INTENT: correction | question | stop | other`, `correction` only if the agent actually said something wrong; parsed by `user.parse_intent`, `other` when missing), recorded on the turn. The prompt is τ-Voice's barge-in prompt (MIT) extended with the persona, the goal and the listening style. No call before `min_words` of the agent turn have been heard |
| `decisions` | object | What the decisions of this episode were: `points` (decision points reached) = `boundary` + `fallback`; `too_few_words` (points skipped, < `min_words` heard); `capped` (points where a safety cap withheld `BACKCHANNEL`); `llm_calls`; `LISTEN` / `BACKCHANNEL` / `INTERRUPT` (the model's choices); `rule_interrupts` |
| `random_events` | object | How many **non-subjective** (random) events this episode had: `noise` (events), `noise_occurrences` (sounds in their bursts), `noise_over_user` (noise events that started during the user's own speech), `aside` (a remark to someone nearby), `called_away`, `capped` (asides dropped by `max_asides_per_turn`), `lines_llm` / `lines_fixed` (aside and called-away lines written by the model / taken from the fixed lists). Each event itself is a turn (§4.4) |
| `noise_events` | array | Each noise event: `{t, label, n, level_db, dur_ms, source, clip, over_user_speech}` — its start, label, occurrences in its burst, level relative to nominal speech (jitter included), length, `user` (an involuntary sound of the caller) or `surroundings`, the clip (`file:<path>` or `synthetic`), and whether it started during the user's own speech. Its turn is the `kind: "noise"` turn starting at `t` |
| `behaviors` | object | The online user's habits. **Random** event rates **per minute of the call** (`aside_per_min`: a seeded Poisson process independent of what the agent does, taken up only when socially plausible — an aside due while the user is saying something, about to, or in the middle of a sequence waits until it is free; `noise_per_min`: noise event onsets, the total of `noise_process`), `noise_process` (the noise event process, below: `model`, `onsets_per_min`, `jitter_db`, `labels: {label: {rate_per_min, source, level_db, burst_p, burst_max, gap_ms, mean_burst}}`), `aside_lines` (`{type: "llm", llm, params}`: the lines of asides and of being called away are written by the user LLM; `{type: "fixed"}`: the fixed lists), `away_p` (the probability that an aside while the agent talks is the user being called away), the **safety caps** (`min_gap_ms` between two user sounds, `max_backchannels_per_turn` per agent turn, `max_asides_per_turn` asides per agent turn (noise events are never capped), `max_pauses_per_turn` mid-thought pauses per user turn; nets against runaway behaviour, not the source of it — `decisions.capped` / `random_events.capped` count how often they held), and `persona`: the profile levels in effect (table below; `inferred` lists those not given in the profile but inferred from `speaking_style` / `traits`, `attentiveness` and whether the scenario has a background). Precedence: `task.scenario["behaviors"]` > the component's constructor values > the profile (constructor `None`, as in `user.PERSONA`). Deprecated: `backchannel_per_min` (mapped to the nearest `backchanneling` level), `pause_p` (a random mid-reply split, off by default; recorded when set), the per-check `*_p` |
| `turn_taking` | object | Effective turn-taking parameters (defaults plus overrides from `task.scenario["turn_taking"]`, e.g. `{"yield_after_ms": null}` for a user who never yields when talked over), in milliseconds: `respond_after_ms` (how long after the agent finishes the user responds), `nudge_after_ms` (how long after its own turn with no response the user speaks again), `max_decision_gap_ms` (the fallback interval between decision points, see `listening`), `check_ms` (how soon the user looks again after yielding), `reaction_ms` (from the decision to barge in to starting to speak), `yield_after_ms` (how long the user keeps talking when the agent talks over it), `speaks_first`. When `response_delay` is present, the user's response gap is not the fixed `respond_after_ms` but random and content-dependent: log-normal (`median_ms`, `sigma`), with probability `distracted_p` of being distracted (`distracted_median_ms`), plus `pause_short_ms` / `pause_long_ms` when the user LLM marked `(pause)` / `(long pause)`. The profile's `attentiveness` (low/normal/high) scales the distraction probability |

Profile levels that control these habits (all optional; inferred when omitted):

| Profile field | Level → behavior | When omitted |
|---|---|---|
| `backchanneling` (the listening style) | told to the listener model as a description (`user.BACKCHANNELING`): `none` never makes listening sounds (`BACKCHANNEL` is not offered) / `rare` / `normal` (now and then, when the agent finishes a point) / `frequent`. The model decides; the level only informs | `speaking_style` / `traits` contains backchannel, mm-hmm, uh-huh, while listening, chatty, talkative → `frequent`; shy, quiet, gruff, terse, curt, clipped, short answers, reserved, or `attentiveness: low` → `rare`; otherwise `normal` |
| `surroundings` (where the user is calling from) | random (asides, noise event onsets) per minute of the call: `quiet` (0, 0.1) / `home` (0.3, 0.2) / `office` (0.2, 0.2) / `car` (0.1, 0.4) / `cafe` (0.2, 0.5) / `street` (0.1, 0.8); asides ×2 when `attentiveness: low`. Asides only make sense when other people are nearby. Also sets which noise events happen how often (per-label rates, table below), the situations of asides (`user.SITUATIONS`), and the default background (§4.2) | The text mentions street/road/outside/traffic/walking → `street`; car/driving → `car`; cafe/coffee shop → `cafe`; office/at work/desk/colleague → `office`; home/kids/children/family/baby/dog → `home` (checked in this order); the scenario has background noise (`scenario["background"]`) → `cafe`; otherwise `quiet` |
| `hesitancy` (stopping mid-utterance to think) | told to the user LLM as a description (`user.HESITANCY`), which places `(pause)` / `(long pause)` markers inside its turn where the person would stop; the turn is then said in pieces (§4.4) | pause, hesitat, thinks out loud, finds words, slow → `high`; fast, quick, precise, clipped, brisk, businesslike, confident → `low`; otherwise `normal` |

**Noise events: a compound Poisson process** (`soundscape.EventProcess`, seeded, on the call clock). Onsets of each label are Poisson at that label's rate for the surroundings (`soundscape.EVENT_RATES`, onsets per minute; `Behaviors.noise_per_min` scales the whole table to a total, `Behaviors.noise_rates` replaces it). Each onset is a burst: after every occurrence another follows with probability `burst_p`, at most `burst_max` in all (a geometric burst size), after a silent gap drawn uniformly from `gap_ms`; the burst is one clip repeated (the same dog, the same phone), cut short where it would pass a per-label length (`soundscape.MAX_BURST_S`: a recording may already hold several barks or rings; `n` in `noise_events` is what was played), and is **one** `noise` turn. Its level is the label's `level_db` (the clip's active-part RMS relative to nominal speech, −20 dBFS) plus a uniform jitter of ±6 dB; its length is the real clip's. Events happen whenever they happen: the user's own involuntary sounds (`source: user`: cough, sneeze, throat clearing) may overlap its own speech and are then part of its audio, still a separate `noise` turn expecting `ignore`; sounds of the surroundings overlap whoever is talking. No safety cap applies to them, and they never delay the user's own lines.

| Surroundings | Onsets per minute, by label |
|---|---|
| `quiet` | cough 0.06, throat_clear 0.03, sneeze 0.01 |
| `home` | cough 0.04, sneeze 0.01, door_slam 0.04, dog_bark 0.04, phone_ring 0.03, dishes 0.04 |
| `office` | cough 0.05, sneeze 0.01, phone_ring 0.05, door_slam 0.03, keyboard 0.06 |
| `car` | cough 0.05, horn 0.15, indicator 0.15, siren 0.05 |
| `cafe` | cough 0.05, dishes 0.15, cup 0.2, door_slam 0.05, phone_ring 0.05 |
| `street` | cough 0.05, horn 0.35, siren 0.12, dog_bark 0.18, door_slam 0.1 |

| Label | Source | Level (dB re speech) | Burst: `burst_p`, `burst_max`, gap (ms) | Mean occurrences |
|---|---|---|---|---|
| cough | user | −2 | 0.4, 3, 250–700 | 1.56 |
| sneeze | user | 0 | 0.15, 2, 400–1200 | 1.15 |
| throat_clear | user | −8 | — | 1 |
| door_slam | surroundings | −10 | — | 1 |
| dog_bark | surroundings | −12 | 0.45, 4, 300–1500 | 1.74 |
| phone_ring | surroundings | −14 | 0.7, 5, 1500–3000 | 2.77 |
| dishes | surroundings | −16 | 0.4, 4, 300–1500 | 1.62 |
| cup | surroundings | −18 | 0.3, 3, 400–2000 | 1.39 |
| keyboard | surroundings | −20 | 0.3, 3, 300–1500 | 1.39 |
| horn | surroundings | −12 | 0.45, 4, 150–900 | 1.76 |
| siren | surroundings | −18 | — | 1 |
| indicator | surroundings | −22 | — | 1 |

**Asides and being called away.** Being addressed stays a Poisson process on the call clock (`aside_per_min`), taken up only when socially plausible (not while the user is speaking, about to, or mid-sequence). A situation is drawn from the episode seed (`user.SITUATIONS[surroundings]`: "their child asks them for something", "the barista calls out their order", ...) and the user LLM writes the lines in one greedy, memoised call (`user.AsideWriter`): the remark to the person nearby and, when called away, the "hold on" to the agent and the "I'm back" — in the persona's language and voice, for the place, the situation and the call so far (at most 12 words a line). A missing or unusable line, or no LLM (a scripted user), falls back to the fixed lists (`user.ASIDES` / `HOLD_ON` / `BACK`). The labels are unchanged (§4.4).

**Subjective vs. random.** Everything the user *decides* — backchannels, barge-ins, mid-thought pauses — is decided by a model in context, never rolled: the listener model at the agent's phrase boundaries, the user LLM inside its turns; the persona levels only describe the person to them. Only what *happens to* the user is random: noise events (involuntary sounds such as a cough count as random) and being addressed or called away by someone nearby, at rates set by the surroundings. Model decisions are greedy (temperature 0) and memoised by prompt; random events come from the episode seed. A barge-in's intent is chosen by the user LLM while it writes the line (see `listening`), not by the listener, so it is `correction` only when the agent got something wrong. The event rates are conservative estimates for phone calls: most calls have no aside or noise; with calls of 1–2 minutes, the chance of at least one event is about 10–20% in a quiet place and 60–80% on the street or in a cafe. For reference, Switchboard telephone conversations have "uh-huh"-type backchannels as about 19% of dialogue acts (Jurafsky et al. 1997), and English listeners backchannel less than Japanese or Mandarin listeners (Clancy et al. 1996); a listener model given a `normal` style should produce a few per minute of the agent's speech, placed at phrase ends.

Run overhead (TTS calls, wall time) belongs to the run, not to the episode, and is not written into the trajectory; only the user's own decision counts (`decisions`, `random_events`) are, since they describe how the episode's user behaved.

### 4.2 `background`

Each item is a background audio track, mixed on top of the user-side audio.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `media` | media reference | yes | See §5 |
| `gain_db` | number | no | Mixing gain, default 0 |
| `loop` | boolean | no | Whether to loop the track when it is shorter than the timeline, default `false` |
| `start_time` / `end_time` | integer | no | Used when the track covers only part of the timeline; covers the whole timeline by default |
| `spec` | object | no | What the track is (below) |

A background track is continuous ambient sound with **no expected reaction**: it is context, never a turn. Discrete noise events (a cough, a door slam) are turns with `kind: "noise"` so that they can be judged individually.

**The episode's background** is part of its environment context and is also recorded, without media, in `meta.env.background` (so it is known even for silence or a text-only run). `UserSim` lays it from `task.scenario["background"]`, or by default from the persona's `surroundings` (`soundscape.background_spec`):

| Spec field | Meaning |
|---|---|
| `type` | `silence` / `white` / `pink` / `brown` / `ambience:<home\|office\|car\|cafe\|street>` |
| `level_dbfs` | RMS level of the track in dBFS |
| `snr_db` | The same as a ratio to nominal speech (`soundscape.SPEECH_DBFS` = −20 dBFS): `snr_db = −20 − level_dbfs` |
| `source` | `none` (silence), `synthetic:<color>` (generated, seeded loop), `file:<path>` (a recording) |
| `from` | `surroundings` (the default) or `scenario` |
| `surroundings` | The persona's surroundings it was chosen for |

Defaults by surroundings: `quiet` → a pink-noise floor at −60 dBFS (`soundscape.QUIET_FLOOR_DBFS`, 40 dB below speech: a phone microphone's self-noise and room tone; a real line is never digital silence; `"background": false` or `"silence"` turns it off); `home` −52, `office` −48, `car` −40, `cafe` −40, `street` −36 dBFS (SNR 32 to 16 dB). An ambience is a recording from the sound bank, normalised to the level; without one it is a coloured-noise stand-in (pink for home / office / cafe, brown for car / street). A bank is laid out as `<dir>/ambience/<kind>/*.wav` and `<dir>/events/<label>/*.wav` (mono 16-bit WAV). By default (`Soundscape()`, `bank="default"`) it is `$IG_SOUNDBANK`, else `~/.cache/interaction_gym/soundbank`. The DEMAND ambience is fetched there on first use (`interaction_gym.noisebank`, the same code as the script below), unless `IG_SOUNDBANK_FETCH=0`. If the fetch fails, the env warns once and uses the stand-ins. `Soundscape(bank=dir)` uses `dir` as it is, and `Soundscape(bank=None)` or `IG_SOUNDBANK=synthetic` uses the stand-ins only. Noise events are never fetched automatically: they come from the bank's `events/` when it has them (the full bank below), else they are synthetic. Which clip and which offset are drawn from the episode seed, so the same seed and the same bank give the same audio; `source` records the file. `task.scenario["background"]` may be a type string, a dict `{type, level_dbfs | snr_db, file}`, `false` (none), or a number — the old form, the gain in dB of the examples' gaussian noise bed (kept at the same loudness, ≈ −31 dBFS + gain). The track loops from a seeded offset.

**Sound bank from DEMAND and MUSAN.** No audio is shipped. The default bank is the DEMAND ambience alone, fetched on first use (above). `scripts/fetch_noise_banks.py --out DIR` downloads both datasets and builds the full bank, `DIR/bank`, in the layout above (standard library only; resumable); use it with `IG_SOUNDBANK=DIR/bank`:

- **ambience** from [DEMAND](https://zenodo.org/records/1227121) (Thiemann, Ito & Vincent 2013; CC BY-SA 3.0 per the record's own text — the Zenodo metadata field says CC BY 4.0; we follow the stricter, as published by the authors): channel 1 of each 16 kHz recording (5 min each; only `ch01.wav` is fetched from each zip, via HTTP range requests), RMS-normalised to −30 dBFS (the env sets the level). Mapping: `home` ← DKITCHEN, DLIVING, DWASHING; `office` ← OOFFICE, OMEETING, OHALLWAY; `cafe` ← PCAFETER (cafeteria), PRESTO (restaurant); `street` ← STRAFFIC, SPSQUARE; `car` ← TCAR. Not used: the nature recordings (NFIELD, NPARK, NRIVER), PSTATION, TBUS, TMETRO (not a car), SCAFE (48 kHz only).
- **events** from [MUSAN](https://www.openslr.org/17/) noise (Snyder, Chen & Povey 2015; OpenSLR SLR17, CC BY 4.0): each clip of `musan/noise/{free-sound,sound-bible}` is labelled by rules on its annotation (`EVENT_RULES`; a reviewed `--labels` JSON overrides them); clips matching no rule, several labels, or music / speech / crowds are left out. A kept clip is trimmed to its active part (the frames within 30 dB of its loudest, at most a per-label length) and normalised to an active-part RMS of −20 dBFS. Labels without a clip keep their synthetic stand-in.
- **coverage** (built 2026-10-07; 116 MB): ambience home 3, office 3, cafe 2, street 2, car 1 recordings (5 min each); events phone_ring 23, siren 17, horn 11, dog_bark 8, cup 8, keyboard 7, dishes 6, door_slam 5, sneeze 1 clips. MUSAN has no usable cough, throat clearing or turn-signal clip: those stay synthetic. The sound-bible clips (88) are labelled by title; the 845 free-sound clips have no metadata at all and are labelled by `scripts/noise_labels.json`: proposals of a zero-shot audio-text model (CLAP, `scripts/label_musan_noise.py`; it agreed with all 8 title-labelled clips) reviewed by hand on spectrogram contact sheets (164 proposed, 79 kept). Cup / dishes are the least certain labels (kitchen clinks look alike).
- all mono 16-bit 16 kHz; `DIR/bank/manifest.json` lists each file with its source file, dataset, license, annotation and the rule that mapped it, plus counts per label / kind.

Other open datasets that would fit (check each license; several are non-commercial):

| Dataset | Fits | License (as published; verify) |
|---|---|---|
| DNS Challenge noise (Microsoft) | both: ~180 h from AudioSet, Freesound, DEMAND | per source; see the repository's license files |
| FSD50K | `events/*`: cough, door, dog bark, telephone, car horn, siren, dishes, typing | per clip (CC0 / CC BY / CC BY-NC) |
| ESC-50 | `events/*`: 40 clips each of dog, door knock, cough, car horn, siren, keyboard, clock alarm, ... | CC BY-NC 3.0 |
| WHAM! noise | `ambience/cafe`, `ambience/street`: urban restaurant / cafe / bar noise | CC BY-NC 4.0 |

### 4.3 `turns`

Sorted by `start_time` ascending (ties keep their order of appearance). A turn is one continuous stretch of sound from one role.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Unique within the episode |
| `role` | string | yes | `user` / `agent` |
| `start_time` | integer | yes | Start instant |
| `end_time` | integer | yes | **Actual** end instant (the cut-off instant if interrupted) |
| `text` | string | yes | The text **actually spoken**; empty string when there is no text (e.g. noise) |
| `media` | media reference | when there is audio or video | References **the segment actually played** |
| `kind` | string | no | Type of this sound, see §4.4; omitted for a normal turn |
| `unsaid` | string | no | The part that was planned but not spoken; **present only when interrupted**. An agent that streams output chunk by chunk has no planned content and never has this field |
| `words` | array | no | Word-level timestamps; each item is `[word, start ms, end ms]`, relative to this turn's `start_time` |
| `tool_calls` | array | no | Tool calls attached to this turn, see §4.5 |
| `expects` | string | no | Overrides the default expected reaction (rare); one of `respond` / `yield` / `continue` / `ignore` / `interrupt` / `wait` |
| `intent` | string | no | A barge-in's purpose, labelled by its producer: `correction` / `question` / `stop` / `other` (from the listener model's decision); not used by the rule scores |
| `label` | string | no | What a sound is, labelled by its producer: a noise's event (`cough`, `door_slam`, `phone_ring`, `dog_bark`, `horn`, ...), or `called_away` on each turn of being called away |
| `to` | string | no | Whom this utterance is addressed to; needed only in multi-party conversations |

Without `words`, "the text spoken up to a given instant" is mapped **linearly** in time (`text` is truncated at the proportion `(t - start_time) / (end_time - start_time)`).

### 4.4 `kind`: type annotated by the producer

Only the **simulation node** that produces a sound (user simulator, a simulator playing some role, the world) writes `kind`; the policy under evaluation does not annotate its own turns.

| `kind` | Meaning | Expected reaction of the other party (fixed rule) |
|---|---|---|
| omitted (normal turn) | Normal speech addressed to the other party | The other party should listen while it is spoken; if the other party is speaking when it starts, the other party should **yield**; after it ends, the other party should respond |
| `backchannel` | Backchannel ("mm-hmm", "right") | The other party **continues**: does not yield and does not answer it as a question |
| `aside` | Speech not addressed to the other party (talking to someone nearby, "hold on") | **Ignore**: keep talking if speaking, stay silent if not |
| `noise` | Non-verbal sound (cough, "uh", background voices); `text` is empty, `label` says what it is | **Ignore**, as above |
| `away` | The user is gone for a while (called away); silence, `text` empty, its duration is the absence | **Wait**: say nothing, or at most briefly check in ("Are you still there?") |

`expects` can override the default rule and is written by the producer: `yield` (the user barges in while the agent is speaking and expects it to yield; set automatically when the user simulation barges in on its own), `wait` (the user pauses mid-utterance and will continue, or asks the agent to hold on; the other party is expected to wait and not take the floor), `interrupt` (a long-winded utterance that the other party is expected to interrupt).

**Every user behaviour is a turn, labelled when it is generated.** `UserSim` makes each of its behaviours a turn of its own (with id, times, text if any, media if any) and labels it at generation time, so `eval.scores` can score it:

| User behaviour | How it is produced | `kind` / `expects` / `intent` / `label` | How `eval.scores` scores it (§6.3) |
|---|---|---|---|
| Normal turn | user LLM / script, after the agent stops | — / — | `respond`: not talked over, answered (graded by latency) |
| Barge-in | listener model chooses `INTERRUPT` at a decision point (or a rule); the user LLM writing the line picks the intent | — / `yield` / `correction\|question\|stop\|other` | `yield`: stop (graded by stop latency), then take the floor again after the user; the intent is a label (whether a correction was taken into account needs a content judge) |
| Backchannel | listener model chooses `BACKCHANNEL` at a phrase boundary | `backchannel` / `ignore` | `ignore`: keep talking (stopping is wrong) |
| Mid-thought pause | the user LLM writes `(pause)` / `(long pause)` inside its turn | each piece before a pause: — / `wait`; the last piece: a normal turn | `wait`: not taking the floor during the pause (and stopping, if the agent was talking); the last piece `respond`. A barge-in said in pieces: the first piece keeps `yield` and, trailing off ("..."), is yielded to and then waited through |
| Noise event (cough, door slam, phone ring, horn ...; a burst is one turn) | random: compound Poisson, per-label rates by surroundings; may overlap anyone's speech, the user's own included | `noise` / `ignore` / — / the event | `ignore` (an agent turn after a noise that overlapped the user's own line counts as the answer to that line) |
| Remark to someone nearby | random, rate by surroundings, when plausible; line written by the user LLM | `aside` / `ignore` | `ignore`: no reply to it, keep talking |
| Called away: "Sorry, hold on a second" | random (an aside while the agent talks, with `away_p`) | — / `wait` / — / `called_away` | `wait`: stop talking and do not take the floor |
| … the remark to the other person | | `aside` / `ignore` / — / `called_away` | `ignore` |
| … the absence | | `away` / `wait` / — / `called_away` | `wait`: 1 if the agent stays silent or makes one short check-in (≤ 3 s, ≥ 2 s into the absence), else 0 |
| … "Sorry about that, go on" | | — / — (`yield` if the agent is talking) / — / `called_away` | a normal turn / barge-in |
| Background track | always on, from surroundings | not a turn (§4.2) | none (context) |

### 4.5 `tool_calls`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Call id |
| `name` | string | yes | Tool name |
| `arguments` | object | yes | Arguments |
| `call_time` | integer | yes | Instant the call is made (the tool executes at this instant) |
| `result_time` | integer | no | Instant the result is delivered; omitted when there is no result |
| `content` | string | no | Returned content (JSON text for structured results) |
| `error` | boolean | no | Whether the call failed, default `false` |

**Which turn it is attached to**: the caller's turn (same `role`). If the caller is speaking at call time, it is attached to the utterance in progress; if not, to the caller's next utterance. If the caller never speaks again, a turn with empty `text` and `start_time = end_time = call_time` is created to hold it.

### 4.6 What is not in the trajectory

The trajectory records only environment-level facts (who said what when, tool calls, evaluation). Agent-internal per-chunk / per-unit information (tokens seen and generated by each model unit, unit boundaries, etc.) **is not part of the trajectory**. When needed, the agent adapter saves it as an optional companion file `agent_traces.jsonl`, matched to the trajectory by `episode_id`; its format is in `docs/AGENT_TRACE.md`.

## 5. Media references

```json
{"kind": "audio", "uri": "media/3f/3fa9…e2.wav", "sr": 16000, "start_ms": 0, "end_ms": 2647}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | `audio` / `video` / `image` |
| `uri` | string | yes | Path relative to `meta.media_root` (absolute addresses such as `s3://` may be supported later) |
| `sr` | integer | for audio | Sample rate |
| `start_ms` / `end_ms` | integer | yes | The referenced segment within the file |

- One file can be referenced several times with different segments: a long recording split into several turns; an interrupted turn references the leading part of the original audio, with no separate file needed.
- Because storage is content-addressed, repeated audio (sentences hit in the TTS cache, recordings reused across episodes) is stored once.
- Mixed audio (user speech + noise + background) is **not saved**; when needed it is re-synthesized from the references and `gain_db`.

## 6. `eval`

```json
"eval": {
  "reward": {"total": 1.0, "parts": {"outcome": 1.0}},
  "duplex": {
    "n0": {"behavior": "noise",       "target": "a0", "reaction": "continued"},
    "u2": {"behavior": "backchannel", "target": "a2", "reaction": "continued"},
    "u3": {"behavior": "barge_in",    "target": "a2", "reaction": "yielded", "latency_ms": 300},
    "u4": {"behavior": "aside",       "target": "a3", "reaction": "continued"}
  },
  "scores": {
    "events": [
      {"turn": "u0", "expects": "respond", "outcome": "responded", "latency_ms": 353, "score": 0.851, "target": "a0", "resolved_ms": 3000, "user_start_ms": 0, "user_end_ms": 2647},
      {"turn": "n0", "expects": "ignore", "outcome": "ignored", "score": 1.0, "target": "a0", "resolved_ms": 3800, "user_start_ms": 3400, "user_end_ms": 3800},
      {"turn": "u3", "expects": "yield", "outcome": "yielded", "score": 0.921, "latency_ms": 300, "target": "a2", "resolved_ms": 13500,
       "reply": "a3", "response_ms": 329, "user_start_ms": 11700, "user_end_ms": 13171},
      ...
    ],
    "by_expectation": {"respond": 0.8788, "ignore": 1.0, "yield": 0.9211},
    "total": 0.9368
  }
}
```

### 6.1 `reward`

| Field | Type | Meaning |
|---|---|---|
| `total` | number or `null` | Total reward; `null` when some part cannot be computed (e.g. it needs an LLM judge that was not run) |
| `parts` | object | Per-part rewards; names depend on the task source (e.g. τ-bench's `DB`, `ENV_ASSERTION`, `ACTION`, `COMMUNICATE`; AutomationBench's `task_completed_correctly`, `partial_credit`) |

### 6.2 `duplex`: special duplex behaviors

**Only special behaviors such as interruptions and non-directed sounds are recorded**; normal exchanges (one party finishes, the other responds) are not. Response latency, when needed, can be computed directly from `turns`. Keys are the ids of the turns that trigger the behavior, in chronological order.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `behavior` | string | yes | Behavior type, see the table below |
| `target` | string | no | The affected turn (of the party speaking at the time); omitted when nobody was speaking |
| `reaction` | string | no | The other party's actual reaction |
| `latency_ms` | integer | no | Present only when `reaction` is `yielded`: time from the start of speaking to the other party stopping |

| `behavior` | Detection rule (derived from `turns`) | Expected | `reaction` values |
|---|---|---|---|
| `barge_in` | The `start_time` of a normal turn falls within `[start_time, end_time)` of one of the other party's turns | The other party yields | `yielded`: the other party's turn has `unsaid` (it was cut; a cut caused only by the episode ending does not count), or it stopped before this turn ended and at least 200 ms after this turn started (ending earlier is just finishing by coincidence, too soon to be a reaction); otherwise `kept_talking` |
| `backchannel` / `aside` / `noise` | Determined by `kind` | The other party is unaffected | While the other party is speaking: `continued` (correct) / `stopped` (wrongly yielded; same rule as above, and the agent turn must end within `stop_grace_ms` = 1000 ms after the sound ends) / `responded` (it finished the turn it was speaking, then started a new turn within 2000 ms of the later of the two end points (this turn's and that turn's), with no normal user turn in between; the original turn already answered the user's question, so the new turn counts as a response to this sound, `eval.reply_after`; applies only to `aside` / `noise`: continuing after a backchannel is exactly what the user wants and is recorded as `continued`). While the other party is not speaking: `stayed_silent` (correct) / `responded` (wrongly responded: the other party starts speaking within 2000 ms after this turn ends, with no other normal turn in between, and that speech is not an answer to the user's previous normal turn; it counts as answering that turn when that turn has not yet been answered and the other party starts speaking at most 5000 ms after it ended) |
| `away` | Determined by `kind` | The other party waits | `waited` / `checked_in` (one agent turn of at most 3000 ms, starting at least 2000 ms into the absence) / `talked` (anything more); `target`: the first agent turn in it |
| `agent_interrupt` | The `start_time` of an agent turn falls within a normal user turn and after that turn's start (starting at the same time counts as `barge_in` above; the key is the agent turn id, `target` is the interrupted user turn) | Should not happen by default; expected behavior when the interrupted user turn has `expects: "interrupt"` | omitted |

The expected reaction is fully determined by `behavior` (and `expects`, if present), so it is not stored separately. `behavior` and the turn's `kind` share values for the three non-directed types but mean different things: `kind` is the ground truth given by the producer; `behavior` is the classification derived by evaluation.

### 6.3 `scores`: was the user's expectation met (rule-based scoring)

`duplex` records what happened; `scores` says **whether it was done right**: **one event, with one score in [0, 1], per user turn**. For each user turn, first determine the one thing the user expects the agent to do; the score is 0 if the agent did not do it, and — where doing it means taking over the conversation (answering, stopping for a barge-in) — the latency satisfaction of how fast it did it. The same scores serve both as evaluation metrics and directly as an RL reward (phase one). (Pre-release versions, until 2026-10-08, gave a turn two or three events, e.g. `no_interrupt` + `respond`; the always-satisfied `no_interrupt` inflated the means.)

**Where the expectation comes from** (`eval.expectation`): if the turn has `expects`, that wins (written by the producer); otherwise it follows from `kind`: backchannel, aside, noise → `ignore`; away → `wait`; a normal turn → `respond`, or `yield` if the agent is speaking when it starts. `expects: "wait"` (a piece before a mid-utterance pause, "hold on") → `wait`, also when the agent is speaking.

Latency satisfaction `latency_score(Δ)`, the same curve for `respond` (Δ = the user's end → the agent's start) and `yield` (Δ = the barge-in's start → the agent's stop): a logistic in milliseconds, `s(Δ) = 1 / (1 + exp((Δ − 950) / 100))`. 200–500 ms is full score, 500–700 ms still high, and from about 800 ms it falls off quickly; it is monotone, so a very early reaction (under 200 ms, but after the user ended) also scores ~1.

| Δ (ms) | 0 | 200 | 500 | 700 | 800 | 900 | 1000 | 1200 | 1500 |
|---|---|---|---|---|---|---|---|---|---|
| score | 1.000 | 0.999 | 0.989 | 0.924 | 0.818 | 0.622 | 0.378 | 0.076 | 0.004 |

(Until 2026-10-09 it was a log-normal bump peaking at 200 ms, σ = 1 in log time: 500 ms → 0.66, 1 s → 0.27, 2 s → 0.07, which penalised a normal 500 ms human gap by a third and, for `respond`, also penalised replies faster than 200 ms.)

| Expectation | Turn | Outcome → score |
|---|---|---|
| `respond` | A normal user turn | `talked_over` → 0 if an agent turn starts while the user is still speaking; else `responded` → `latency_score` of the gap from the user's end to the agent's start, if the agent starts after the user finishes, before the next directed user turn and within 5 s; else `no_response` → 0. If the episode ends (`end_ms`) before that window closes, the event is `censored`: no `score`, left out of the means |
| `yield` | A normal user turn that starts while the agent is speaking (a barge-in) | `kept_talking` → 0 if the agent is neither cut nor stops before the user finishes (a cut caused only by the episode end is not a stop); `talked_over` → 0 if it stops but starts again while the user is still speaking; `no_response` → 0 if it stops but does not take the floor again in the `respond` window; else `yielded` → `latency_score` of the stop (from the user's start to the agent's stop). A censored window leaves the stop alone to score. A barge-in piece that trails off ("Actually, wait..."; the user goes on after a pause) is yielded to and then waited through: `took_floor` → 0 if the agent speaks before the user continues |
| `ignore` | Backchannel, aside, noise | `ignored` → 1 if the agent neither stops (it is cut within 1 s of the sound's end) nor replies to the sound; `stopped` / `replied` → 0. A reply right after the agent finished the turn it was speaking counts (`eval.reply_after`, see §6.2); an agent turn after a noise that overlapped the user's own line counts as the answer to that line, not to the noise |
| `wait` | `expects: "wait"`: a piece before a mid-utterance pause, or "hold on" | `kept_talking` → 0 if the agent was speaking and did not stop; `took_floor` → 0 if it starts speaking during the turn or after it before the user's next directed turn; else `waited` → 1. For an absence (`kind: "away"`): `waited` or `checked_in` (one agent turn of at most 3 s, at least 2 s into the absence) → 1, `took_floor` (anything more) → 0 |
| `interrupt` | `expects: "interrupt"` | `cut_in` → 1 if the agent starts speaking during the turn, else `listened` → 0 |

Result: `{"events": [{turn, expects, outcome, score?, latency_ms?, target?, reply?, response_ms?, user_start_ms, user_end_ms, resolved_ms?}], "by_expectation": {expectation: mean score}, "total": mean score over the scored user turns}`; `total` is `null` when nothing was scored. `latency_ms` is the latency that was graded (the reply's for `respond`, the stop's for `yield` and for a `wait` that stopped the agent); a `yield` that was answered also gives the `reply` turn and its `response_ms` (from the user's end).

Each event also carries **position fields** (position only, no effect on the score) so the training side can assign events to a segment of the episode: `user_start_ms` / `user_end_ms` (the user turn), `resolved_ms` (when the outcome was settled: the start of the response / the stop / the agent turn that talked over the user or took the floor, or the end of the window in which nothing happened; absent for `censored`), and, when an agent turn decided the outcome, `target` (the responding turn, the turn that yielded or kept talking, the turn that talked over the user or took the floor, the turn that reacted to the sound). Events without `target` are ones where the agent did nothing (`no_response`, `waited` in silence, `ignored` in silence).

The duplex timing metrics (`eval.timing_counts` / `timing_summary`: turn-take rate, response and yield latency, cut-in rate, collisions) are computed from `turns` with the same rules, independently of the events.

## 7. Relation to the runtime event log

| Direction | Procedure |
|---|---|
| Runtime → storage | Multiple segment events with the same id are merged into one turn: the last version gives `end_time` and `text`; if the first version is longer, the difference is written to `unsaid`. Tool calls and results are paired by id and attached to turns. Whether the maximum duration was reached is written to `meta.end_reason`. |
| Storage → runtime | Each turn is restored as one segment event (plus a truncation event at `end_time` if interrupted); tool calls are restored as call and result events. Can be replayed as an exogenous stream (offline RL, regression tests). |
| Agent view | Slicing the user-side turns into the segments actually played at each step, for a given `chunk_ms`, yields the observations the agent received; turns with `role` agent and tool calls are its actions. |

Information **intentionally dropped** on save:
- The "planned duration" of utterances that were not interrupted (only what actually happened is kept);
- The chunking and per-chunk text of a streaming agent (use `words` when precise alignment is needed);
- Debugging information such as the simulator's internal decisions (e.g. the result of each barge-in check); export it separately when needed.

## 8. Versioning

- Incrementing `schema` signals an incompatible change; adding optional fields does not require an increment.
- Readers should ignore unknown fields.
