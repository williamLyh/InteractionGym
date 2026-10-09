# User-voice A/B (2026-10-08)

Which way of voicing the simulated user sounds most natural and consistent across a call? 30 episodes of a text-only
run (persona user turns, backchannels, asides; text, speaker and order held fixed) re-synthesized under 7 `Voice`
variants, judged by a local audio LLM (Qwen3-Omni-30B-A3B-Instruct, absolute scores + pairwise comparisons in both
orders) and objective metrics (speaking-rate and LUFS spread within a call, ECAPA speaker similarity, Qwen3-ASR WER,
DNSMOS, latency). The winner (neutral reference turn + clone + leveling + per-episode seed) is `Voice`'s default.

`pipeline.sh` runs every stage on one 8-GPU host (set `VOICE_AB_ROOT` to a work directory and `IG_SERVICES` to the serving directory): `synth.py`
(4 CustomVoice + 4 Base TTS replicas) → `bench.py` (latency) → `metrics.py` + `loud.py` → `judge.py` (`run_judge.sh`:
upstream vLLM, TP 4 × DP 2) → `report.py` (report.md / report.json). Results: CHANGELOG (2026-10-08).
