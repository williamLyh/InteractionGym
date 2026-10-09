# Serving the models an episode needs

Closed-loop episodes with real models need up to three kinds of servers, all OpenAI-compatible:

| role | what | protocol | client variable |
|---|---|---|---|
| user simulator: words | a chat LLM (vLLM) | `POST /v1/chat/completions` | `IG_LLM_URL` |
| user simulator: voice | TTS (vLLM-Omni Qwen3-TTS), optionally a voice-clone TTS | `POST /v1/audio/speech` | `IG_TTS_URL`, `IG_CLONE_URL` |
| the agent | a full-duplex model on vLLM-Omni (MiniCPM-o 4.5, ...) | WebSocket `/v1/realtime?duplex=1` | `IG_AGENT_URL` (`IG_AGENT_URLS` for several) |

Any server with these APIs works. [`reference/`](reference/) is an **example deployment** for one host with
8 GPUs of 32 GB (we ran it on RTX 5090s): parameterised scripts that start every service in tmux, with paths and
the GPU layout in one settings file. Treat it as a starting point and adapt it to your hardware.

Input-clocked lockstep (`clock: "input"`), `silence_continuation` and the per-unit token trace need a vLLM-Omni
build with our duplex patches ([docs/agent_server.md](../../docs/agent_server.md)); realtime mode works on stock
vLLM-Omni. To decide how many GPUs each service gets, see the GPU tuner ([docs/GPU_TUNER.md](../../docs/GPU_TUNER.md)).
