# Third-party material

InteractionGym's own code is licensed under [Apache-2.0](LICENSE). This file lists every third-party
source the repository uses, under which license, what exactly we use, and whether the repository redistributes it.

**Benchmark data is never redistributed.** The `benchmarks/` modules are loaders: you download each dataset
yourself and must follow its license (several are non-commercial). The same holds for model weights.

## Code and prompts included in this repository

### In the main package (`interaction-gym`, Apache-2.0 plus these permissive notices)

| Source | License | What we include | Where |
|---|---|---|---|
| [Audio MultiChallenge](https://huggingface.co/datasets/ScaleAI/audiomc) (Scale AI) | MIT | The rubric judge prompt from the dataset card (itself adapted from Arora et al., 2025) | `.../third_party/audiomc/` |
| [tau2-bench / τ-Voice](https://github.com/sierra-research/tau2-bench) (Sierra Research) | MIT | The user simulator's barge-in decision prompt (`INTERRUPT_PROMPT`), adapted; timing-metric definitions followed in `eval.py` | `src/interaction_gym/user.py` |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) | Apache-2.0 | Deploy YAMLs derived from `vllm_omni/deploy/*.yaml` in the reference deployment | `examples/serving/reference/configs/` |

### Separate optional component: `interaction-gym-fdbench` (CC BY-NC 4.0, non-commercial only)

The material ported from Full-Duplex-Bench is **not part of the Apache-2.0 package** (`interaction-gym`):
it is a separate distribution in `extras/fdbench/` with its own `LICENSE` (CC BY-NC 4.0), `NOTICE`, `pyproject.toml`
(`license = "CC-BY-NC-4.0"`) and Python package `interaction_gym_fdbench`. The main wheel and sdist do not
contain it; install it with the extra (`uv sync --extra fdbench`, or `pip install ./extras/fdbench`).

| Source | License | What it contains | Where |
|---|---|---|---|
| [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) v1 / v1.5 (`v1_v1.5/evaluation/`) | **CC BY-NC 4.0** | Ports of the official metric rules (take-turn rule, backchannel TOR / JSD histogram, `get_timing.py` overlaps and response gaps, thresholds), the user-interruption judge prompt and the behaviour judge's input layout | `extras/fdbench/src/interaction_gym_fdbench/v1.py` |
| Full-Duplex-Bench v2 (`v2/`) | **CC BY-NC 4.0** | The judge rubric prompts (`eval/eval_prompts.json`), the judge prompt layout (`eval_single_item.py`), the examinee prompt and examiner suffix (`run_dataset.sh`) | `.../interaction_gym_fdbench/v2.py` |
| Full-Duplex-Bench v3 (`v3/`) | **CC BY-NC 4.0** | The 12 mock APIs (`mock_apis.py`), tool schemas and agent instructions (`cascaded_agent.py`), latency profiles (`latency_injector.py`), argument / response judge prompts and pass logic (`evaluate_pass_rate.py`, `evaluate_tool_calls.py`) | `.../interaction_gym_fdbench/v3.py` |

Each file carries an SPDX header, its upstream source and a description of our changes. The Apache-2.0 modules
`benchmarks.full_duplex_bench`, `benchmarks.fdb2` and `benchmarks.fdb3` (loaders, the closed-loop user, timing)
import without the component and load it lazily on first use of a metric, prompt or tool schema; without it they
raise an `ImportError` that says how to install it. **Installing and using the component makes that use subject to
CC BY-NC 4.0 (non-commercial only).** Nothing else in the package uses it.

## Used at run time, not redistributed

| Source | License | How it is used |
|---|---|---|
| tau2-bench (Python package `tau2`) | MIT | Optional extra `tau`, installed from its git repository at a pinned commit; domain data read from your checkout (`third_party/tau2-bench` or `$TAU2_DATA_DIR`) |
| [AutomationBench](https://github.com/zapier/AutomationBench) (Zapier) | MIT | Imported from your checkout (`third_party/automationbench` or `$AUTOMATIONBENCH_PATH`); tasks, simulator and rubric are not copied. `HANDWRITTEN_SOLUTIONS` are our own |
| Full-Duplex-Bench v1.5 behaviour-judge instruction (`behavior.txt`) and `icc_gt_distribution.json` | CC BY-NC 4.0 | Read from your checkout of the official repository (`--repo`, `--gt`) |
| vLLM-Omni | Apache-2.0 | The agent / TTS server you run; `examples/serving/reference/scripts/make_minicpmo_overlay.py` patches a copy of your installed package locally |
| Optional Python packages: `websockets` (BSD-3-Clause), `pydantic` (MIT), `numpy` (BSD-3-Clause); `pyarrow`, `soundfile` for `audiomc.extract` | permissive | Dependencies, not vendored |

## Datasets (loaders only; never redistributed)

| Dataset | Data license | Loader |
|---|---|---|
| Full-Duplex-Bench v1.0 Candor / ICC subsets | CC BY-NC 4.0 + the upstream corpora's terms | `benchmarks.full_duplex_bench` |
| Full-Duplex-Bench v1.0 synthetic subsets, all of v1.5 | MIT | `benchmarks.full_duplex_bench`, `benchmarks.clip_bank` |
| Full-Duplex-Bench v2 (`prompts_staged_200.json`) | CC BY-NC 4.0 | `benchmarks.fdb2` |
| Full-Duplex-Bench v3 recordings + metadata | CC BY-NC 4.0 | `benchmarks.fdb3` |
| [Easy Turn testset](https://huggingface.co/datasets/ASLP-lab/Easy-Turn-Testset) | Apache-2.0 | `benchmarks.easy_turn` |
| [HumDial-FDBench](https://huggingface.co/datasets/ASLP-lab/HumDial-FDBench) | Apache-2.0 | `benchmarks.humdial`, `benchmarks.clip_bank` |
| Audio MultiChallenge | MIT | `benchmarks.audiomc` |
| tau2-bench domains | MIT | `integrations.tau` |
| AutomationBench public tasks | MIT | `integrations.automationbench` |

## Noise datasets (downloaded on your machine; never redistributed)

**We ship only download / build code, not the audio.** No noise audio is in the repository or in any package.

- **DEMAND (default).** The env downloads DEMAND from its official Zenodo record on first use
  (`interaction_gym.noisebank`) and builds the default ambience bank from it in a user cache
  (`~/.cache/interaction_gym/soundbank`, or `$IG_SOUNDBANK`). `IG_SOUNDBANK=synthetic` turns this off.
- **MUSAN (optional).** `scripts/fetch_noise_banks.py` downloads both datasets and builds the full bank, which adds
  MUSAN noise events.

Each bank's `manifest.json` records the source file, dataset and license of every clip. A bank built on your
machine is yours to use under the datasets' licenses; a DEMAND-derived bank is ShareAlike (CC BY-SA 3.0).

| Dataset | License (checked 2026-10-07 at the official source) | What the script uses | Attribution |
|---|---|---|---|
| [DEMAND](https://zenodo.org/records/1227121) (doi:10.5281/zenodo.1227121) | **CC BY-SA 3.0** Unported, per the record's description ("This work, the audio data and the document describing it, is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License"); the Zenodo metadata field says CC BY 4.0 — we follow the authors' own, stricter statement | channel 1 of 11 of the 16 kHz recordings (ambience), RMS-normalised; the default background, fetched on first use | J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)", ICA 2013. Derived banks are ShareAlike |
| [MUSAN](https://www.openslr.org/17/) (OpenSLR SLR17) | **CC BY 4.0** (the OpenSLR page) | clips of `musan/noise/` with a clear event label, trimmed and normalised (full bank only; never fetched automatically) | D. Snyder, G. Chen, D. Povey, "MUSAN: A Music, Speech, and Noise Corpus", arXiv:1510.08484, 2015; per-clip sources (Freesound, SoundBible) in `musan/noise/*/ANNOTATIONS` and the bank manifest |

The archive is fetched from the OpenSLR mirrors, and from a Hugging Face copy only when its size and sampled bytes
match OpenSLR's.

`benchmarks.clip_bank` cuts clips from data you downloaded into a local directory; by default it extracts only
the MIT / Apache-2.0 subsets (Candor / ICC only with `allow_noncommercial=True`). The clips keep their source
license in the manifest and are not part of this repository.

## Models

No model weights are included. Models referenced in examples and the reference deployment (e.g. Qwen3 LLMs,
Qwen3-TTS, MiniCPM-o 4.5, Nemotron VoiceChat, AURA, PersonaPlex, VoxCPM2, Phi-4-mini, Qwen3-ASR) are
downloaded by you under their own licenses.
