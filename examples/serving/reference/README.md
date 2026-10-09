# Reference deployment (example)

An example of serving everything a closed-loop episode needs on **one Linux host with 8 GPUs of 32 GB**, each
server in its own tmux session, bound to `127.0.0.1`. It is a template, not a turnkey installer: you provide the
Python environments and the model weights; the scripts take paths and the GPU layout from `serving.env`.

```
start.sh stop.sh status.sh        core services: LLM, TTS (+ proxy), optional clone TTS (+ proxy), agent server(s)
start_extras.sh stop_extras.sh    optional extra user-sim models (compatibility experiments)
serving.env.example               every setting with its default: copy to serving.env and edit
common.sh                         settings, caches and helpers sourced by every script
scripts/run_*.sh                  one launcher per server (usable on their own)
scripts/tts_proxy.py              least-in-flight balancer over TTS replicas
scripts/make_minicpmo_overlay.py  optional MiniCPM-o memory overlay (caps Code2Wav's lazy CUDA graphs)
scripts/minicpmo_duplex_client.py, lockstep_check.py, token_trace_check.py   protocol smoke tests
configs/*.yaml                    vLLM-Omni deploy configs (stage -> GPU placement, memory shares)
patches/*.patch                   MiniCPM-o fixes for the vLLM-Omni install: per-session feature extractor,
                                  Thinker-only text sessions (the default agent layout)
pyshim/                           import alias MiniCPM-o's Code2Wav needs (on PYTHONPATH for the agent server)
```

## Setup

1. Copy this directory to the GPU host (say `/opt/dig_services`), `cp serving.env.example serving.env`.
2. Python environments (paths in `serving.env`; each a directory with `bin/vllm`):
   - `envs/llm`: [vLLM](https://github.com/vllm-project/vllm) for the user-simulator LLM;
   - `envs/tts`: [vLLM-Omni](https://github.com/vllm-project/vllm-omni) for TTS (stock is fine), with `fastapi`,
     `httpx` and `uvicorn` for the TTS proxy;
   - `envs/omni`: vLLM-Omni for the duplex agents. For lockstep and the token trace, install the build with our
     duplex patches (upstream PR [vllm-project/vllm-omni#8485](https://github.com/vllm-project/vllm-omni/pull/8485);
     until it is merged: fork `williamLyh/vllm-omni`, branch `duplex-input-clock`). With a stock build set
     `AGENT_TOKEN_TRACE=0` and run the client in realtime mode. MiniCPM-o also needs the `stepaudio2-minicpmo`
     package (Token2wav). Apply both patches to that install:
     ```bash
     cd <envs/omni>/lib/python3.12/site-packages
     patch -p1 < <this dir>/patches/minicpmo_fe_per_session.patch
     patch -p1 < <this dir>/patches/minicpmo_thinker_only.patch
     ```
     (`patch -p1 --dry-run` first to check; to keep the install untouched, apply them to an overlay copy instead,
     e.g. one built by `scripts/make_minicpmo_overlay.py`, and set `IG_MINICPMO_OVERLAY`.)
     - `minicpmo_fe_per_session.patch`: upstream MiniCPM-o duplex sessions share one audio feature extractor whose
       log-mel floor each session's stream rewrites, so concurrent sessions change each other's input normalisation
       (also the reference-voice features of a new session).
     - `minicpmo_thinker_only.patch`: a session whose output modalities exclude audio ends at Stage 0 (the
       Thinker; no Talker / Code2Wav work, the transcript comes from the Thinker's tokens), and the
       `minicpmo_4_5_thinker` pipeline for the one-GPU Thinker-only layout. Audio sessions are unchanged, so the
       same install serves both layouts. Not upstream; the header of the patch has the details and measurements.
   No system CUDA toolkit? The launchers use the pip one inside the environment (`nvidia/cu13`) for flashinfer's JIT.
3. Model weights as local directories under `models/` (or `IG_MODELS_DIR`), named as in `serving.env`:
   `Qwen3.8-27B-FP8` (LLM), `Qwen3-TTS-12Hz-1.7B-CustomVoice` (TTS), `Qwen3-TTS-12Hz-1.7B-Base` (voice clone;
   the simulated user clones its first turn by default), `MiniCPM-o-4_5` (agent). Download them from Hugging Face or ModelScope under their licenses.
4. Edit the layout in `serving.env` (`LLM_GPUS`, `TTS_REPLICAS`, `CLONE_REPLICAS`, `AGENT_LAYOUT`, `AGENT_SERVERS`,
   ports).

## Run

```bash
./start.sh       # (re)start the core services; prints the IG_* variables for clients
./status.sh      # sessions, endpoint health, GPU memory
./stop.sh        # stop them (also stops servers started by the GPU tuner's launch.sh)
tmux capture-pane -pt dig_llm | tail      # a live log; full logs in logs/
```

Default layout (`serving.env.example`): LLM TP2 on GPUs 0-1 (`:8000`), one TTS replica on each of 6 and 7 behind
the proxy (`:8001`), and four Thinker-only MiniCPM-o servers, one per remaining GPU (2-5 on `:8010`-`:8013`, 16
sessions each). Clients get all four in `IG_AGENT_URLS`.

### Agent layouts

| `AGENT_LAYOUT` | GPUs per server | config | sessions | serves | use |
|---|---|---|---|---|---|
| `thinker` (default) | 1 | `configs/minicpmo_4_5_thinker_1gpu.yaml` | 16 | text-only sessions only | evaluation and RL rollouts: the runners' default (text timed at `speech_cps`) |
| `audio` | 2 (Thinker \| Talker + Code2Wav) | `configs/minicpmo_4_5_2gpu.yaml` | 4-6 | audio and text-only sessions | `--audio-out` runs: the agent's real speech, recordings, demos |

The Thinker-only layout needs `patches/minicpmo_thinker_only.patch` (Setup, step 2). A client of it must ask for
text only (`VllmOmniDuplexAgent(audio_out=False)`, the runners' default): an audio session gets no output there.
Its results are not bit-identical to the audio layout's (bf16-level logprob differences from the first speak unit,
so sampled trajectories diverge), its agent timing is estimated from the text, and they are not directly comparable
with `--audio-out` runs (docs/BENCHMARKS.md, "Agent output").

```bash
# serving.env, the default: four Thinker-only servers
AGENT_LAYOUT=thinker
AGENT_SERVERS="2:8010 3:8011 4:8012 5:8013"
AGENT_MAX_SESSIONS=16          # about 8 for Audio MultiChallenge (long episodes share the KV cache)
# for --audio-out runs instead: one audio server on GPUs 4-5 (two: "0,1:8010 2,3:8011" with LLM_GPUS=4,5)
AGENT_LAYOUT=audio
AGENT_SERVERS="4,5:8010"
AGENT_MAX_SESSIONS=4
```

One server by hand: `AGENT_LAYOUT=thinker GPUS=3 PORT=8011 scripts/run_minicpmo.sh` (or `AGENT_LAYOUT=audio
GPUS=4,5 ...`). The layout the GPU tuner found fastest for audio lockstep episodes on such a host was two audio
servers: `AGENT_SERVERS="0,1:8010 2,3:8011"`, `LLM_GPUS=4,5`, `TTS_REPLICAS="6:8002"`,
`CLONE_REPLICAS="7:8030 6:8031"`, `AGENT_MAX_SESSIONS=6`.

Startup takes a few minutes (LLM ~3-5 min, TTS ~2 min, MiniCPM-o ~5 min). On the audio layout, the very first
duplex session after a start can return listen decisions only (no audio) while the Talker / Code2Wav finish lazy
initialisation: send one throw-away session first, e.g. `python scripts/minicpmo_duplex_client.py basic --mode fast`.

From another machine, tunnel the ports: `ssh -N -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8010:localhost:8010 <gpu-host>`.

## Service notes

**LLM** (`scripts/run_llm.sh`): thinking is off by default (`--default-chat-template-kwargs {"enable_thinking": false}`);
send `"chat_template_kwargs": {"enable_thinking": true}` to enable it per request (reasoning goes to
`message.reasoning`). Default sampling is Qwen's recommended non-thinking setting (temperature 0.7, top_p 0.8,
top_k 20, presence_penalty 1.5); request fields override it. FP8 weights and KV cache, `--max-model-len 32768`,
text only (`--language-model-only`).

**TTS** (`scripts/run_tts.sh`, ~12 GB per replica): `POST /v1/audio/speech` with
`{"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "input", "voice", "response_format": "pcm"}` returns raw 16-bit
mono PCM at 24 kHz. Voices (`GET /v1/audio/voices`): aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu,
vivian. Optional `language` and `instructions` (style / emotion).

**Voice clone** (Qwen3-TTS Base, `CLONE_REPLICAS`): `task_type: "Base"`, `ref_audio` as a data URL
(`data:audio/wav;base64,...`) or `file://` URI, `ref_text` its transcript (or `x_vector_only_mode: true`).

**MiniCPM-o 4.5** (`scripts/run_minicpmo.sh`): one server per GPU (Thinker-only) or GPU pair (audio); duplex
sessions cannot be data-parallel across API servers, so throughput comes from more servers. Audio input only
(`limit_mm_per_prompt: {video: 0, image: 1}`: the shipped video profiling runs out of memory on 32 GB).
Thinker-only (`configs/minicpmo_4_5_thinker_1gpu.yaml`): Stage 0 alone, `max_num_batched_tokens: 4096`,
`max_model_len: 16384` (a session grows ~16 tokens per second of conversation; Audio MultiChallenge episodes reach
~15k), 16 sessions; Stage-0 sampling as in the audio config. The KV cache (~60k tokens on a 32 GB card) is shared by
the sessions: enough for 16 FD-Bench-length episodes, about 8 for Audio MultiChallenge.
Audio (`configs/minicpmo_4_5_2gpu.yaml`): Thinker on the first GPU, Talker and Code2Wav on the second. On 32 GB cards
Code2Wav shares the second GPU with the Talker; the config lowers the Talker's memory share and caps Code2Wav's CUDA
graphs so that 4-6 concurrent audio sessions fit. If Code2Wav still runs out of memory, build the overlay
(`scripts/make_minicpmo_overlay.py`, `hift_max_lazy_graphs: 0`) and set `IG_MINICPMO_OVERLAY`.
Protocol, lockstep and token trace: [docs/agent_server.md](../../../docs/agent_server.md).

**Other duplex agents** (`scripts/run_duplex_{nemotron,aura,personaplex}.sh`; compatibility table in
docs/agent_server.md): their configs are filled in with your model and vLLM-Omni paths at launch.

## Extra user-sim models (optional)

`./start_extras.sh` / `./stop_extras.sh` start models for trying other user-simulator combinations (edit GPUs and
memory shares in the script for your layout):

| port | model | notes |
|---|---|---|
| 8006 | Qwen3-TTS-12Hz-1.7B-VoiceDesign | `task_type: "VoiceDesign"`, voice described in `instructions`; a new voice per request |
| 8007 | Qwen3-TTS-12Hz-0.6B-CustomVoice | same API as the main TTS |
| 8008 | VoxCPM2 | 48 kHz output; needs the `voxcpm` package (`VOXCPM_PYTHONPATH`) |
| 8020 | Qwen3-4B | vLLM, no reasoning parser: send `chat_template_kwargs: {"enable_thinking": false}` |
| 8021 | Phi-4-mini-instruct | vLLM |

## GPU tuner

`python -m interaction_gym.gpu_tuner` searches the layout ([docs/GPU_TUNER.md](../../../docs/GPU_TUNER.md)).
Its host presets in `src/interaction_gym/gpu_tuner/hosts/` launch the scripts of this directory:
`example-8gpu-thinker.toml` (Thinker-only agent servers, one GPU each; `examples/gpu_tuner.py`'s default) and
`example-8gpu.toml` (audio agent servers, two GPUs each; `--audio-out`). Set `workdir` to where you copied it. The tuner writes and runs its own `launch.sh` (sessions
`dig_agent0`, `dig_llm`, ...); `./stop.sh` stops those too.
