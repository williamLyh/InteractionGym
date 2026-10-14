# A second duplex model: Gander vs MiniCPM-o 4.5, open vs closed loop

Does the closed loop change how two agents compare, or only the absolute numbers? The release run
([BENCHMARKS.md](BENCHMARKS.md), 2026-10-09) has one agent. This adds **Gander** ([Gander-Omni/Gander](https://huggingface.co/Gander-Omni/Gander),
arXiv 2609.08977, Apache-2.0): a full-duplex SFT of MiniCPM-o 4.5 with its own cadence (8 text tokens per 1 s unit,
an `<|interrupt|>` token). Same base model, different post-training. Run 2026-10-14 on one host.

## Setup

The release protocol, unchanged except for the agent:

- **Agent.** `--agent gander` (env `b79672f`). Thinker-only text output timed at Gander's own speaking rate, 12.8
  characters per second. The rate was calibrated with `agents.speech_rate` on 34 complete English turns from 29
  lockstep audio FD-Bench v3 episodes; MiniCPM-o's is 11.3. Lockstep, greedy decoding, prompts identical to `minicpmo`.
- **Serving.** Three one-GPU servers, 16 sessions each. The overlay is the release's `eval_thinker` (Thinker-only patch
  plus the per-session feature extractor) plus a port of the Gander cadence patch (reads the checkpoint's
  `duplex_cadence`).
  - That serving patch is not in `patches/` yet.
  - One server crashed twice (vLLM engine `Invalid request status: RUNNING`). The watchdog restarted it, and the
    runners redid the lost episodes.
- **Data, seeds, users, judges.** Identical to the release: the same recordings, the same FD-Bench v2 static scripts
  (byte-identical copies) and the same Audio MultiChallenge ids. The user, examiner and judges are all local
  Qwen3.8-27B (proxy judges).
- **FD-Bench v2 seeds.** Gander ran seeds 0–1 only: the seed-2 scripts could not be copied between the hosts. Every
  FD-Bench v2 comparison below uses seeds 0–1 for both models.

**Consistency.** A and B of Gander agree before the user's second turn at the noise floor of two A runs:

| | A vs B | A vs A, two seeds |
|---|---|---|
| FD-Bench v3, first response, same timing | 98.2% | 98.2% |
| FD-Bench v3, same fulfilment | 90.0% | 91.8% |
| FD-Bench v2, same text | 340/400 | — |

## Results

Paired per (item, seed). The last three columns give the Gander − MiniCPM-o difference in each condition, and the
change of that gap from open to closed loop. 95% bootstrap CIs. Script: [`compare_models.py`](compare_models.py).

| metric | n | MiniCPM-o A | MiniCPM-o B | Gander A | Gander B | Gander − MiniCPM-o, A | Gander − MiniCPM-o, B | gap change (B − A) |
|---|---|---|---|---|---|---|---|---|
| FD-Bench v3 spoken fulfilment (A vs B final) | 500 | 0.822 | 0.984 | 0.820 | 0.962 | −0.002 [−0.040, +0.038] | **−0.022 [−0.042, −0.002]** | −0.020 [−0.058, +0.016] |
| FD-Bench v3 slot outcome | 500 | 0.890 | 0.991 | 0.899 | 0.983 | +0.009 [−0.017, +0.036] | −0.008 [−0.020, +0.003] | −0.018 [−0.043, +0.008] |
| FD-Bench v3 response quality | 500 | 0.290 | 0.854 | 0.514 | 0.854 | **+0.224 [+0.170, +0.278]** | +0.000 [−0.036, +0.034] | **−0.224 [−0.288, −0.162]** |
| FD-Bench v2 IF | 397 | 3.90 | 4.34 | 3.38 | 3.91 | −0.52 [−0.62, −0.43] | −0.43 [−0.53, −0.34] | +0.09 [−0.02, +0.21] |
| FD-Bench v2 TT | 397 | 4.09 | 4.42 | 3.49 | 4.15 | −0.59 [−0.68, −0.51] | −0.27 [−0.35, −0.19] | **+0.32 [+0.21, +0.44]** |
| FD-Bench v2 task score | 300 | 3.79 | 4.38 | 3.41 | 4.02 | −0.38 [−0.54, −0.22] | −0.35 [−0.51, −0.20] | +0.02 [−0.18, +0.22] |
| Audio MultiChallenge rubric items | 200 | 0.241 | 0.285 | 0.244 | 0.272 | +0.002 [−0.042, +0.048] | −0.013 [−0.057, +0.032] | −0.015 [−0.079, +0.047] |
| Audio MultiChallenge all items | 200 | 0.065 | 0.075 | 0.075 | 0.080 | +0.010 [−0.035, +0.055] | +0.005 [−0.035, +0.040] | −0.005 [−0.065, +0.055] |

Gander's own reports (FD-Bench v2 at 180 s, plus the 120 s window) are on 5090-2 in `/home/work/ig_gander/runs/`.

## What the second model shows

1. **The closed loop can reverse a comparison.**
   - FD-Bench v3 response quality: in the open loop, Gander's first answer is judged better than MiniCPM-o's by 22
     points (0.51 vs 0.29). In the closed loop the two tie at 0.854.
   - FD-Bench v3 spoken fulfilment shows the reverse: the open loop ties the two models (0.820 vs 0.822), and only the
     closed loop separates them, MiniCPM-o ahead by 2.2 points with a CI that excludes 0.
   - Which agent "wins" depends on the protocol.
2. **The open loop exaggerates a turn-taking deficit.** On FD-Bench v2, Gander's TT gap to MiniCPM-o halves in the closed
   loop (−0.59 → −0.27; the gap change is +0.32, CI excluding 0).
   - The static script never waits for the agent, and it collides with Gander more often: 4.20 collisions per
     conversation (74% of examiner lines) against MiniCPM-o's 3.59 (63%).
   - Those collisions are scored as turn-taking failures of the agent, but they are artefacts of the replayed script.
3. **Where nothing depends on the agent's previous turn, the closed loop changes nothing.** Audio MultiChallenge (recorded turns
   replanned to the agent) leaves both models at the same level and the same gap, as it did in the release run. FD-Bench v2 IF and
   task score shift both models alike.
4. **Behaviour seen only in the closed loop:** claiming a result the agent cannot know (it has no tools).
   - Gander: A 17%, B 71%. MiniCPM-o: A 5%, B 53%.
   - Both models invent completed actions once a caller follows up.

## Limits

- Two agents that share a base model. One ranking flip on one metric is evidence that open-loop comparisons can
  mislead, not an estimate of how often they do.
- All judges are proxies (Qwen3.8-27B for GPT-4o / Gemini / o4-mini). FD-Bench v3 response quality in B judges the
  whole conversation, A only the first response.
- Text-only output timed at a calibrated rate, not real speech: timing-sensitive numbers (TT, collisions) carry its
  error of a few tenths of a second per utterance.
