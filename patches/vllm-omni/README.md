# vLLM-Omni patch for InteractionGym

`VllmOmniDuplexAgent` needs a few vLLM-Omni server features that are not upstream yet. This directory has them as
one patch per supported vLLM-Omni base: install vLLM-Omni, apply one patch, and the server has every feature
InteractionGym uses. When an upstream PR is merged and released, the matching part is removed from the patch.

| file | for |
|---|---|
| `vllm_omni-0.31.0rc1-interactiongym.patch` | **vllm-omni 0.31.0rc1 from PyPI** (with `vllm==0.31.0`), applied to site-packages. Also applies to a checkout of tag `v0.31.0rc1`. Package files only. |
| `vllm_omni-main-61cae20d-interactiongym.patch` | a **source checkout of vllm-project/vllm-omni at commit `61cae20d0f5c2b438564a892e250e08e72e5afa2`** (main, 2026-10-09; `vllm==0.31.0`). Package files plus the upstream unit tests and docs of every part. |
| `components/0N-*.patch` | the six parts below, package files only, in apply order. The combined patch for 0.31.0rc1 is these six applied in order. They are for reading and for dropping a part once it is upstream. |

Both combined patches make the same change to `vllm_omni/` (27 files); only the context lines differ.

## Which base, and why

- **vllm-omni 0.31.0rc1 (PyPI), the default.** It is the newest release on PyPI, so `pip install vllm-omni==0.31.0rc1`
  followed by the patch is all a user needs, and it is 34 commits behind the commit the PRs are based on. All six
  parts apply to it without fuzz. It is a pre-release; pip installs it only when the version is pinned exactly, as below.
- **main at `61cae20d`, for source checkouts.** The PRs live on upstream main. That commit contains both PR bases:
  `b21df3bc` for #8485 and its follow-ups, and `9cc105d1` for #8638.
- **Not supported: vllm-omni 0.30.0** (the last non-rc release, vLLM 0.30). Between 0.30.0 and the PR base,
  upstream changed the duplex engine in about 25 commits. These include the duplex protocol module, bounded output delivery
  (#7644), full-duplex `/v1/realtime` (#7285), PersonaPlex's move into the framework (#7695), MiniCPM-o's KV window
  (#7821) and the vLLM 0.31 rebase (#8459). `git apply --check` of the patch reports 14 errors there. Making it work would mean
  re-implementing the PRs on an older engine rather than porting them, so no 0.30.0 patch is shipped. 0.30.0 also lacks
  #8227 (see docs/agent_server.md, "MiniCPM-o 4.5 protocol notes").

## What is inside

| part | what it adds | upstream | remove when |
|---|---|---|---|
| 01 `input-clock` | `extra_body.silence_continuation` (`false`: the server never feeds the model silence of its own); input-clocked sessions, `extra_body.clock: "input"`: model time advances only with client input, and every append / commit / `response.create` gets one `input_audio_buffer.processed` after all of its outputs; refused inputs are acknowledged (`decision: "rejected"`, `reason`), so the adapter can resend them; per-unit `units[]` with `listen` / `speak` / `timed_out` ...; the unit timeouts; the `DuplexModelPlugin` hooks (`supports_input_clock`, `unit_decision`, `unit_output_complete`) | [vllm-project/vllm-omni#8485](https://github.com/vllm-project/vllm-omni/pull/8485) (open), head `eacc510a` = fork `williamLyh/vllm-omni` branch `duplex-input-clock` | #8485 is merged and released |
| 02 `token-trace` | opt-in per-unit token trace: `extra_body.trace_tokens: true` (input-clocked sessions only) sends one `debug.unit_tokens` per unit with the ids each stage consumed and produced; the server must enable it with `duplex_session.enable_debug_events: true`; MiniCPM-o 4.5 reports its Thinker / Talker ids | follow-up of #8485, not opened yet (fork branch `duplex-input-clock-pr2-v3`) | that PR is merged |
| 03 `minicpmo-input-clock` | MiniCPM-o 4.5 opts into the input clock: units end where Code2Wav's audio for them ends (`tts_is_last_chunk`), `speak_empty` units, backlog submission | follow-up of #8485, not opened yet (`duplex-input-clock-pr3a-v3`) | that PR is merged |
| 04 `qwen3-omni-input-clock` | Qwen3-Omni opts in (commit-based turns); unit hooks for Nemotron VoiceChat, PersonaPlex and AURA, which do **not** opt in yet (see below) | follow-up of #8485, not opened yet (`duplex-input-clock-pr3b-v3`) | that PR is merged |
| 05 `minicpmo-fe-per-session` | each MiniCPM-o duplex session gets its own audio feature extractor (the shared one's log-mel floor leaked between sessions and into the next session's reference voice) | [vllm-project/vllm-omni#8638](https://github.com/vllm-project/vllm-omni/pull/8638) (open), head `2bde5dc0` | #8638 is merged and released |
| 06 `minicpmo-thinker-only` | a session whose output modalities exclude audio ends at Stage 0 (the Thinker): no Talker / Code2Wav work, the transcript comes from the Thinker's tokens; pipeline `minicpmo_4_5_thinker` (Stage 0 only, one GPU) for `examples/serving/reference/configs/minicpmo_4_5_thinker_1gpu.yaml`. Generic part: `DuplexModelPlugin.text_output_stage_id` (opt-in per plugin). Audio sessions are unchanged | InteractionGym only, not proposed upstream | kept until upstream has a text-only duplex path |

Parts 01-04 are taken unchanged from the latest revision of the PR stack: #8485 after review (8 commits), with
the token-trace, MiniCPM-o and other-model follow-ups rebased onto it. Only two doc paragraphs had to be merged by
hand, because 03 and 04 are sibling branches. Part 06 was ported to that revision from the earlier fork patch
(`fp/0006`). It also gives a text-only unit a proper `units[].decision`, `speak_empty` or `listen` where the
Thinker gave the Talker nothing, instead of a plain `speak`.

What InteractionGym relies on, and the part that provides it: lockstep acknowledgements and resends of refused
inputs (01, 03), `silence_continuation: false` (01), the token trace `debug.unit_tokens` (02, 03), the per-session
feature extractor (05), and Thinker-only text sessions and the one-GPU layout (06).

### Not included, or different from earlier builds

- **Input clock for Nemotron VoiceChat, PersonaPlex and AURA.** Their hooks are in part 04, but they do not opt in,
  as in the PR: each still needs an engine prerequisite. On those models, a `clock: "input"` session is refused with
  `input_clock_unsupported`; run them with `clock="realtime"`. The prototype build on our hosts
  (`0.30.1.dev97+ge7c7dac58`) enabled them.
- **The token trace needs the input clock.** The PR refuses `trace_tokens` without `clock: "input"`
  (`token_trace_requires_input_clock`); the prototype allowed it. `VllmOmniDuplexAgent` therefore requests the trace
  only in lockstep (`trace_tokens=True` with `clock="realtime"` warns and records nothing).
- **`units[]` format.** A timed-out unit is `{"decision": "timed_out", "reason": "no_progress" | "max_age"}` (the
  prototype sent `"timed_out": true`).
- **Not shipped:** the mid-turn `<|listen|>` mask (`extra_body.mask_midturn_listen`, fork patch `fp/0007`). It is
  used only by RL training, not by InteractionGym.

## Apply

The script finds the `vllm_omni` package, checks that its version or commit is a supported base, dry-runs the patch,
backs up every file it touches, and applies it. It needs GNU `patch` (`apt-get install patch`; macOS:
`brew install gpatch`).

**pip install (site-packages):**

```bash
python3.12 -m venv ~/envs/omni
~/envs/omni/bin/pip install vllm==0.31.0 vllm-omni==0.31.0rc1
~/envs/omni/bin/pip install stepaudio2-minicpmo        # MiniCPM-o's Token2wav (audio output)
~/envs/omni/bin/pip install nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88   # see "Install pitfalls"
c=~/envs/omni/lib/python3.12/site-packages/nvidia/cu13; ln -s lib $c/lib64; ln -s libcudart.so.13 $c/lib/libcudart.so
python3 scripts/apply_vllm_omni_patch.py --python ~/envs/omni/bin/python --dry-run
python3 scripts/apply_vllm_omni_patch.py --python ~/envs/omni/bin/python
```

By hand, from the environment's site-packages:

```bash
cd ~/envs/omni/lib/python3.12/site-packages
patch -p1 --dry-run < <repo>/patches/vllm-omni/vllm_omni-0.31.0rc1-interactiongym.patch
patch -p1 < <repo>/patches/vllm-omni/vllm_omni-0.31.0rc1-interactiongym.patch
```

**Source checkout:**

```bash
git clone https://github.com/vllm-project/vllm-omni && cd vllm-omni
git checkout 61cae20d0f5c2b438564a892e250e08e72e5afa2
pip install vllm==0.31.0 && pip install -e .
python3 <repo>/scripts/apply_vllm_omni_patch.py --target .        # or: --python <env>/bin/python
# by hand: patch -p1 --dry-run < <repo>/patches/vllm-omni/vllm_omni-main-61cae20d-interactiongym.patch, then without --dry-run
```

A checkout of tag `v0.31.0rc1` takes the 0.31.0rc1 patch; the script picks it.

**Undo:** `python3 scripts/apply_vllm_omni_patch.py --python <env>/bin/python --revert` restores the touched files byte
for byte and deletes the files the patch added. The backup lives in `<site-packages or checkout>/.interaction_gym_vllm_omni_patch/`.
Reinstalling vLLM-Omni also drops the patch. After that, delete the backup directory, or the next apply refuses to run.

## Install pitfalls

- **flashinfer's JIT fails at server start** with `CUDA compiler and CUDA toolkit headers are incompatible`. A fresh
  resolve of `vllm==0.31.0` installs `nvidia-cuda-nvcc` / `nvidia-nvvm` 13.4 next to the CUDA 13.0 runtime headers, and flashinfer
  0.7 compiles its sampling kernels with that nvcc on first use. Pin the compiler to the runtime:
  `pip install nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88` (nvvm too: a 13.4 `cicc` emits PTX 9.4, which the 13.0 `ptxas` rejects). The launchers of the serving reference use the
  pip CUDA in the environment (`nvidia/cu13`) when there is no system toolkit. The JIT also links against
  `nvidia/cu13/lib64/libcudart.so`, which the wheels do not provide. Without a system toolkit, add two links
  (`ld: cannot find -lcudart` otherwise):
  `c=<env>/lib/python3.12/site-packages/nvidia/cu13; ln -s lib $c/lib64; ln -s libcudart.so.13 $c/lib/libcudart.so`.
- **A reinstall or upgrade of vllm-omni silently drops the patch**, and `--status` then reports `not-applied`. Delete
  `<site-packages>/.interaction_gym_vllm_omni_patch/` and apply again.
- **macOS:** the built-in BSD `patch` is not supported; the script needs GNU patch (`brew install gpatch`).

## Check that it is applied

```bash
python3 scripts/apply_vllm_omni_patch.py --python <env>/bin/python --status    # "...: applied", exit code 0
<env>/bin/python -c 'from vllm_omni.engine.duplex.session import input_clock, token_trace
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin as P
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
assert P.supports_input_clock and P.text_output_stage_id == 0 and "minicpmo_4_5_thinker" in OMNI_PIPELINES; print("patched")'
```

Against a running MiniCPM-o server (audio layout), `examples/serving/reference/scripts/lockstep_check.py` and
`token_trace_check.py` check the protocol end to end, and `examples/minicpmo_agent.py --clock input --trace-tokens`
runs a full episode. An unpatched server ignores `clock: "input"` and never sends `input_audio_buffer.processed`, so
the adapter fails with its `ack_timeout_s` error.

### On main at 61cae20d: the reference deploy configs need three changes on 32 GB cards

Main captures MiniCPM-o's Stage-0 encoder CUDA graphs after the memory profile (#8430) and profiles larger vision
tiles (#8462). With `examples/serving/reference/configs/minicpmo_4_5_*.yaml` as they are (tuned on 0.31.0rc1),
Stage 0 runs out of memory at start-up on an RTX 5090. The verification ran main with three changes:
- Stage 0 `enforce_eager: true`;
- `limit_mm_per_prompt.image: 0`;
- in the audio layout, Stage 0 `max_model_len: 16384`.

None of this is needed on 0.31.0rc1.

## Verified

2026-10-09 on 5090-2 (8x RTX 5090 32 GB, driver 580 / CUDA 13.0), in fresh venvs:
- `pip install vllm==0.31.0 vllm-omni==0.31.0rc1`, patched in site-packages;
- `vllm==0.31.0` plus `pip install -e` of a checkout at `61cae20d`, patched in the checkout.

Each patch was applied with `scripts/apply_vllm_omni_patch.py` (dry run, apply, `--status`, a second apply reports
"already applied"). In site-packages, the 27 patched files were identical to the same patch applied to a `v0.31.0rc1`
checkout.

- **Upstream unit tests.** These ran before and after the patch: `tests/engine/duplex`,
  `tests/worker/test_native_duplex_hooks.py` and
  `tests/model_executor/stage_input_processors/test_minicpmo_4_5_async_chunk.py`, including the new tests of every part.
  - 0.31.0rc1: 618 passed before, 817 passed after.
  - main: 624 passed before, 823 passed after.
  - In both, 0 failed, 1 skipped and 1 xfailed.
- **MiniCPM-o 4.5, both layouts of `examples/serving/reference`, on each base.** Thinker-only ran on one GPU and audio
  on two, next to the reference LLM, TTS and clone TTS.
  - `scripts/lockstep_check.py` (audio layout): all four checks pass, (a) (b) (c) (d0). No events and no model time
    while input pauses, 4.6-6.3x faster than real time with correct answers, identical event sequences with and
    without random 0-2 s client sleeps, and realtime sessions unaffected.
  - `scripts/token_trace_check.py` (audio layout): all 8 checks pass.
  - `examples/minicpmo_agent.py --clock input --trace-tokens` ran three ways: against the Thinker-only server,
    against the audio server with `--audio-out`, and against the audio server text-only (Thinker-only sessions on
    the two-GPU layout). Each episode was lockstep, 40-60 s of conversation in 9-16 s of wall time. The agent spoke
    4-6 turns, 40-61 units were traced, with no retries and no dropped inputs.
  - `examples/minicpmo_suite.py` against the Thinker-only server: 6 episodes, one of them realtime and untraced.
    Every episode has agent turns, and the 5 lockstep episodes have their traces.
- **`--revert`.** 1,826 files of the rc1 site-packages `vllm_omni` were byte-identical to before the patch. The main
  checkout was byte-identical with a clean `git status`.
